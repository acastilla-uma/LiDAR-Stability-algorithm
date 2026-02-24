"""
Unified Terrain Provider

Abstraction layer selecting between LAZ and TIF data sources.
Automatically chooses the most appropriate source for a given coordinate.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from .laz_reader import LAZReader
from .tif_reader import TIFReader

logger = logging.getLogger(__name__)


class TerrainProvider:
    """
    Unified interface for terrain data (elevation, roughness).
    Automatically selects LAZ or TIF source based on coordinate location.
    """
    
    def __init__(self, laz_dir: Optional[str] = None, tif_path: Optional[str] = None):
        """
        Initialize terrain provider.
        
        Args:
            laz_dir: Directory containing .laz files (e.g., LiDAR-Maps/cnig/)
            tif_path: Path to background TIF file (e.g., LiDAR-Maps/geo-mad/dtm.tif)
        """
        self.laz_readers = {}  # LAZ filename -> LAZReader object
        self.tif_reader = None
        self.priority = 'laz'  # Prefer LAZ over TIF when both available
        
        # Load TIF if provided
        if tif_path:
            try:
                self.tif_reader = TIFReader(tif_path)
                logger.info(f"Loaded TIF: {Path(tif_path).name}")
            except Exception as e:
                logger.warning(f"Failed to load TIF: {e}")
        
        # Discover and load LAZ files
        if laz_dir:
            laz_dir = Path(laz_dir)
            if not laz_dir.exists():
                logger.warning(f"LAZ directory not found: {laz_dir}")
            else:
                laz_files = sorted(laz_dir.glob("*.laz"))
                logger.info(f"Found {len(laz_files)} LAZ files in {laz_dir.name}")
                
                for laz_file in laz_files:
                    try:
                        self.laz_readers[laz_file.name] = LAZReader(str(laz_file))
                        logger.debug(f"  ✓ {laz_file.name}")
                    except Exception as e:
                        logger.warning(f"  ✗ Failed to load {laz_file.name}: {e}")
        
        if not self.laz_readers and not self.tif_reader:
            raise ValueError("No LAZ or TIF data source provided")
        
        logger.info(f"Terrain provider initialized: {len(self.laz_readers)} LAZ + " +
                   ("1 TIF" if self.tif_reader else "0 TIF"))
    
    def _find_containing_laz(self, x: float, y: float) -> Optional[str]:
        """
        Find which LAZ file contains the point.
        
        Args:
            x: UTM X coordinate
            y: UTM Y coordinate
            
        Returns:
            LAZ filename or None
        """
        for filename, reader in self.laz_readers.items():
            xmin, ymin, xmax, ymax = reader.get_bounds()
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return filename
        return None
    
    def get_elevation(self, x: float, y: float, source: str = 'auto') -> Optional[float]:
        """
        Get elevation at a point.
        
        Args:
            x: UTM X coordinate
            y: UTM Y coordinate
            source: 'auto' (try LAZ first, fallback TIF), 'laz', 'tif', or 'both'
            
        Returns:
            Elevation value in meters or None
        """
        if source == 'auto' or source == 'laz':
            # Try LAZ
            laz_file = self._find_containing_laz(x, y)
            if laz_file:
                reader = self.laz_readers[laz_file]
                patch = reader.extract_patch(x, y, radius_m=1.0)
                if len(patch) > 0:
                    return float(np.mean(patch[:, 2]))
            
            # Fallback to TIF
            if source == 'auto' and self.tif_reader:
                return self.tif_reader.get_elevation(x, y, interpolate=True)
        
        elif source == 'tif' and self.tif_reader:
            return self.tif_reader.get_elevation(x, y, interpolate=True)
        
        elif source == 'both':
            # Average both sources
            elevations = []
            
            # Try LAZ
            laz_file = self._find_containing_laz(x, y)
            if laz_file:
                reader = self.laz_readers[laz_file]
                patch = reader.extract_patch(x, y, radius_m=1.0)
                if len(patch) > 0:
                    elevations.append(np.mean(patch[:, 2]))
            
            # Try TIF
            if self.tif_reader:
                z = self.tif_reader.get_elevation(x, y, interpolate=True)
                if z is not None:
                    elevations.append(z)
            
            if elevations:
                return float(np.mean(elevations))
        
        return None
    
    def extract_terrain_patch(self, x: float, y: float, 
                             radius_m: float = 50.0, 
                             source: str = 'auto') -> Optional[np.ndarray]:
        """
        Extract terrain patch (3D point cloud or 2D raster).
        
        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius_m: Patch radius in meters
            source: 'auto', 'laz', 'tif', or 'both'
            
        Returns:
            For LAZ: (N, 3) array of [X, Y, Z]
            For TIF: (M, M) array of elevations
            For 'both': dict with 'laz' and 'tif' keys
        """
        if source == 'auto' or source == 'laz':
            # Try LAZ
            laz_file = self._find_containing_laz(x, y)
            if laz_file:
                reader = self.laz_readers[laz_file]
                patch = reader.extract_patch(x, y, radius_m)
                if len(patch) > 0:
                    return patch
            
            # Fallback to TIF
            if source == 'auto' and self.tif_reader:
                patch_size = int(2 * radius_m)
                return self.tif_reader.extract_patch(x, y, patch_size)
        
        elif source == 'tif' and self.tif_reader:
            patch_size = int(2 * radius_m)
            return self.tif_reader.extract_patch(x, y, patch_size)
        
        elif source == 'both':
            result = {}
            
            # LAZ patch
            laz_file = self._find_containing_laz(x, y)
            if laz_file:
                reader = self.laz_readers[laz_file]
                result['laz'] = reader.extract_patch(x, y, radius_m)
            
            # TIF patch
            if self.tif_reader:
                patch_size = int(2 * radius_m)
                result['tif'] = self.tif_reader.extract_patch(x, y, patch_size)
            
            return result if result else None
        
        return None
    
    def get_coverage_info(self) -> Dict:
        """
        Get information about data coverage.
        
        Returns:
            Dict with coverage statistics
        """
        laz_bounds = [reader.get_bounds() for reader in self.laz_readers.values()]
        
        info = {
            'n_laz_files': len(self.laz_readers),
            'n_laz_points': sum(len(reader.points) for reader in self.laz_readers.values()),
            'has_tif': self.tif_reader is not None
        }
        
        if laz_bounds:
            xs = [b[0] for b in laz_bounds] + [b[2] for b in laz_bounds]
            ys = [b[1] for b in laz_bounds] + [b[3] for b in laz_bounds]
            info['laz_coverage_area'] = (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        if self.tif_reader:
            bounds = self.tif_reader.get_bounds()
            info['tif_coverage_area'] = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        
        return info
    
    def is_point_covered(self, x: float, y: float, source: str = 'auto') -> bool:
        """
        Check if a point has data coverage.
        
        Args:
            x: UTM X coordinate
            y: UTM Y coordinate
            source: 'auto', 'laz', 'tif', or 'both'
            
        Returns:
            True if point is covered
        """
        if source == 'laz' or source == 'auto':
            if self._find_containing_laz(x, y):
                return True
        
        if (source == 'tif' or source == 'auto') and self.tif_reader:
            bounds = self.tif_reader.get_bounds()
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                return True
        
        return False
    
    def close_all(self):
        """Close all file handles."""
        for reader in self.laz_readers.values():
            if hasattr(reader, 'close'):
                reader.close()
        if self.tif_reader:
            self.tif_reader.close()
