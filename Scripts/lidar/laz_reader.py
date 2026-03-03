"""
LAZ/LAS Point Cloud Reader

Reads compressed point clouds from CNIG PNOA LiDAR data.
Filters for ground points (ASPRS class 2).
Provides spatial indexing for fast patch extraction.
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LAZReader:
    """
    Reader for .laz compressed point cloud files.
    Supports ASPRS classification filtering and spatial indexing.
    """
    
    def __init__(self, laz_path: str, filter_ground=True):
        """
        Initialize LAZ reader.
        
        Args:
            laz_path: Path to .laz file
            filter_ground: If True, keep only ASPRS class 2 (ground)
        """
        self.filepath = Path(laz_path)
        if not self.filepath.exists():
            raise FileNotFoundError(f"LAZ file not found: {laz_path}")
        
        # Load the point cloud
        logger.info(f"Loading LAZ file: {self.filepath.name}")
        self.las = laspy.read(str(self.filepath))
        
        logger.info(f"  Total points: {len(self.las)}")
        logger.info(f"  CRS: {self.las.header.offsets}")
        logger.info(f"  Z range: [{self.las.z.min()}, {self.las.z.max()}]")
        
        # Filter if requested
        if filter_ground:
            self._filter_ground()
        
        # Extract coordinates
        self.points = np.vstack([self.las.x, self.las.y, self.las.z]).T
        
        # Build spatial index in XY only (2D), not XYZ
        logger.info(f"Building KD-tree with {len(self.points)} points...")
        self.kdtree = cKDTree(self.points[:, :2])
        
        logger.info(f"LAZ reader initialized: {len(self.points)} ground points indexed")
    
    def _filter_ground(self):
        """Filter to keep only ground points (ASPRS class 2)."""
        # ASPRS classification: 2 = ground
        ground_mask = self.las.classification == 2
        
        n_ground = np.sum(ground_mask)
        n_total = len(self.las)
        
        logger.info(f"  Filtering ground points: {n_ground}/{n_total} ({100*n_ground/n_total:.1f}%)")
        
        if n_ground == 0:
            logger.warning("  No ground points found! Using all points.")
            return
        
        # Apply filter
        self.las = self.las[ground_mask]
    
    def get_bounds(self):
        """
        Get bounding box of point cloud.
        
        Returns:
            (xmin, ymin, xmax, ymax)
        """
        return (
            self.points[:, 0].min(),
            self.points[:, 1].min(),
            self.points[:, 0].max(),
            self.points[:, 1].max()
        )
    
    def extract_patch(self, x_center: float, y_center: float, radius_m: float) -> np.ndarray:
        """
        Extract point cloud patch around a location.
        
        Args:
            x_center: Center X coordinate (UTM)
            y_center: Center Y coordinate (UTM)
            radius_m: Search radius in meters
            
        Returns:
            Array of shape (N, 3) containing [X, Y, Z] for N points
        """
        # Query KD-tree for points within XY radius (2D)
        center = np.array([x_center, y_center])
        indices = self.kdtree.query_ball_point(center, radius_m)
        
        if len(indices) == 0:
            logger.debug(f"No points found in patch at ({x_center:.0f}, {y_center:.0f}) with radius {radius_m}m")
            return np.array([]).reshape(0, 3)
        
        # Filter by XY distance only
        patch = self.points[indices]
        xy_dist = np.sqrt((patch[:, 0] - x_center)**2 + (patch[:, 1] - y_center)**2)
        patch = patch[xy_dist <= radius_m]
        
        return patch
    
    def extract_patch_knn(self, x_center: float, y_center: float, k: int) -> np.ndarray:
        """
        Extract K nearest neighbors around a location.
        
        Args:
            x_center: Center X coordinate
            y_center: Center Y coordinate
            k: Number of neighbors
            
        Returns:
            Array of shape (K, 3) containing [X, Y, Z]
        """
        center = np.array([x_center, y_center])
        distances, indices = self.kdtree.query(center, k=k)
        
        return self.points[indices]
    
    def get_crs(self):
        """Get coordinate reference system info."""
        # For PNOA data, typically ETRS89 UTM
        return {
            'srs_name': self.las.header.parse_crs(),
            'scale': self.las.header.scales,
            'offset': self.las.header.offsets
        }
    
    def get_stats(self):
        """Get point cloud statistics."""
        return {
            'n_points': len(self.points),
            'x_range': (self.points[:, 0].min(), self.points[:, 0].max()),
            'y_range': (self.points[:, 1].min(), self.points[:, 1].max()),
            'z_range': (self.points[:, 2].min(), self.points[:, 2].max()),
            'z_mean': float(self.points[:, 2].mean()),
            'z_std': float(self.points[:, 2].std())
        }
