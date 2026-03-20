"""
Terrain Feature Extraction

Computes terrain-derived features:
- φ_lidar: Transverse topographic slope
- TRI: Terrain Roughness Index (standard deviation of elevation)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict
from scipy.signal import convolve

logger = logging.getLogger(__name__)

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=float)


class TerrainFeatureExtractor:
    """
    Computes terrain-derived stability features from elevation patches.
    """
    
    @staticmethod
    def compute_slope_aspect(dem: np.ndarray, resolution: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute slope and aspect from a DEM.
        
        Uses Sobel operators for gradient estimation.
        
        Args:
            dem: 2D elevation array
            resolution: Pixel size in meters
            
        Returns:
            (slope_rad, aspect_rad) as 2D arrays
        """
        # Handle NaN values
        dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
        
        scale = 8 * resolution
        dz_dx = convolve(dem_filled, SOBEL_X / scale, mode='same')
        dz_dy = convolve(dem_filled, SOBEL_Y / scale, mode='same')
        
        # Compute slope and aspect
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(dz_dy, dz_dx)
        
        return slope, aspect
    
    @staticmethod
    def compute_phi_lidar(dem: np.ndarray, vehicle_track: float = 2.48,
                         resolution: float = 1.0) -> float:
        """
        Compute transverse topographic slope φ_lidar.
        
        Transverse slope = cross-slope perpendicular to vehicle motion.
        Approximated as the mean of 90-degree rotated terrain slope.
        
        Args:
            dem: 2D elevation array
            vehicle_track: Vehicle track width in meters (DOBACK024: 2.48m)
            resolution: Pixel size in meters
            
        Returns:
            φ_lidar in radians
        """
        # Alternative: Extract cross-slope profile
        # For a 2D patch, the cross-slope at center is approximated
        center_row = dem.shape[0] // 2
        cross_profile = dem[center_row, :]
        
        # Fit line to cross_profile and compute slope
        if len(cross_profile) > 1 and not np.all(np.isnan(cross_profile)):
            x = np.arange(len(cross_profile)) * resolution
            cross_profile_valid = np.nan_to_num(cross_profile, nan=np.nanmean(cross_profile))
            
            coeffs = np.polyfit(x, cross_profile_valid, 1)
            return float(np.arctan(coeffs[0]))

        # Fallback when the cross profile is unusable.
        slope, _ = TerrainFeatureExtractor.compute_slope_aspect(dem, resolution)
        return float(np.mean(np.abs(slope)))
    
    @staticmethod
    def compute_tri(dem: np.ndarray) -> float:
        """
        Compute Terrain Roughness Index (TRI).
        
        TRI = sqrt(mean((Z_ij - Z_center)^2))
        
        Measures elevation variability; higher TRI = rougher terrain.
        
        Args:
            dem: 2D elevation array
            
        Returns:
            TRI value in meters
        """
        # Handle NaN
        dem_valid = dem[~np.isnan(dem)]
        
        if len(dem_valid) == 0:
            return 0.0
        
        z_mean = np.mean(dem_valid)
        tri = np.sqrt(np.mean((dem_valid - z_mean)**2))
        
        return float(tri)
    
    @staticmethod
    def compute_terrain_ruggedness_index(dem: np.ndarray, 
                                        resolution: float = 1.0) -> float:
        """
        Alternative roughness metric: mean absolute elevation difference.
        
        More robust to outliers than TRI.
        
        Args:
            dem: 2D elevation array
            resolution: Pixel size in meters
            
        Returns:
            Ruggedness index in meters
        """
        # Fill NaN values before computing differences
        dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
        
        if dem_filled.size < 4:
            return 0.0
        
        h_diffs = np.abs(np.diff(dem_filled, axis=1)).ravel()
        v_diffs = np.abs(np.diff(dem_filled, axis=0)).ravel()
        all_diffs = np.concatenate((h_diffs, v_diffs))
        valid = all_diffs[np.isfinite(all_diffs)]

        if valid.size == 0:
            return 0.0

        return float(np.mean(valid))
    
    @staticmethod
    def compute_elevation_stats(dem: np.ndarray) -> Dict[str, float]:
        """
        Compute basic elevation statistics.
        
        Args:
            dem: 2D elevation array
            
        Returns:
            Dict with z_min, z_max, z_mean, z_std
        """
        dem_valid = dem[~np.isnan(dem)]
        
        if len(dem_valid) == 0:
            return {
                'z_min': np.nan,
                'z_max': np.nan,
                'z_mean': np.nan,
                'z_std': np.nan,
                'z_range': np.nan
            }
        
        return {
            'z_min': float(np.min(dem_valid)),
            'z_max': float(np.max(dem_valid)),
            'z_mean': float(np.mean(dem_valid)),
            'z_std': float(np.std(dem_valid)),
            'z_range': float(np.max(dem_valid) - np.min(dem_valid))
        }
    
    @staticmethod
    def extract_features(dem: np.ndarray, 
                        vehicle_track: float = 2.48,
                        resolution: float = 1.0) -> Dict[str, float]:
        """
        Extract all terrain features from a DEM patch.
        
        Args:
            dem: 2D elevation array
            vehicle_track: Vehicle track width in meters
            resolution: Pixel size in meters
            
        Returns:
            Dict with keys:
            - 'phi_lidar': Transverse topographic slope (rad)
            - 'tri': Terrain Roughness Index (m)
            - 'ruggedness': Alternative roughness metric (m)
            - 'z_min': Minimum elevation (m)
            - 'z_max': Maximum elevation (m)
            - 'z_mean': Mean elevation (m)
            - 'z_std': Elevation std dev (m)
            - 'z_range': Elevation range (m)
        """
        # Handle all-NaN patches
        if np.all(np.isnan(dem)):
            logger.warning("All-NaN DEM patch encountered")
            return {
                'phi_lidar': 0.0,
                'tri': 0.0,
                'ruggedness': 0.0,
                'z_min': np.nan,
                'z_max': np.nan,
                'z_mean': np.nan,
                'z_std': np.nan,
                'z_range': np.nan
            }
        
        features = {
            'phi_lidar': TerrainFeatureExtractor.compute_phi_lidar(dem, vehicle_track, resolution),
            'tri': TerrainFeatureExtractor.compute_tri(dem),
            'ruggedness': TerrainFeatureExtractor.compute_terrain_ruggedness_index(dem, resolution)
        }
        
        # Add elevation stats
        stat_keys = ['z_min', 'z_max', 'z_mean', 'z_std', 'z_range']
        elev_stats = TerrainFeatureExtractor.compute_elevation_stats(dem)
        for key in stat_keys:
            features[key] = elev_stats[key]
        
        return features
    
    @staticmethod
    def extract_features_from_point_cloud(points: np.ndarray,
                                        vehicle_track: float = 2.48) -> Dict[str, float]:
        """
        Extract terrain features from a 3D point cloud.
        
        For point clouds, we approximate by gridding to a DEM first.
        
        Args:
            points: (N, 3) array of [X, Y, Z]
            vehicle_track: Vehicle track width in meters
            
        Returns:
            Dict with terrain features
        """
        if len(points) < 4:
            logger.warning("Point cloud too small for feature extraction")
            return {
                'phi_lidar': 0.0,
                'tri': 0.0,
                'ruggedness': 0.0,
                'z_min': np.nan,
                'z_max': np.nan,
                'z_mean': np.nan,
                'z_std': np.nan,
                'z_range': np.nan
            }
        
        # Create gridded DEM from point cloud
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Define grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Use 1m grid resolution
        x_grid = np.arange(x_min, x_max, 1.0)
        y_grid = np.arange(y_min, y_max, 1.0)
        
        # Interpolate points to grid
        from scipy.interpolate import griddata
        xx, yy = np.meshgrid(x_grid, y_grid)
        points_xy = np.vstack([x, y]).T
        dem = griddata(points_xy, z, (xx, yy), method='nearest')
        
        # Extract features from gridded DEM
        return TerrainFeatureExtractor.extract_features(dem, vehicle_track, resolution=1.0)
