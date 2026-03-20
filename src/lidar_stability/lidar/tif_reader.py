"""
GeoTIFF Raster Reader

Reads Digital Terrain Model (DTM) from GeoTIFF files.
Provides elevation extraction by UTM coordinates.
"""

import numpy as np
import rasterio
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TIFReader:
    """
    Reader for GeoTIFF raster files (Digital Terrain Models).
    Provides coordinate transformation and elevation queries.
    """
    
    def __init__(self, tif_path: str):
        """
        Initialize TIF reader.
        
        Args:
            tif_path: Path to .tif GeoTIFF file
        """
        self.filepath = Path(tif_path)
        if not self.filepath.exists():
            raise FileNotFoundError(f"TIF file not found: {tif_path}")
        
        # Open the raster
        logger.info(f"Loading TIF file: {self.filepath.name}")
        self.src = rasterio.open(str(self.filepath))
        
        logger.info(f"  Size: {self.src.width} x {self.src.height} pixels")
        logger.info(f"  CRS: {self.src.crs}")
        logger.info(f"  Transform: {self.src.transform}")
        
        # Read entire raster into memory for faster access
        self.data = self.src.read(1)  # Read first band (elevation)
        
        logger.info(f"  Z range: [{np.nanmin(self.data):.1f}, {np.nanmax(self.data):.1f}]")
        logger.info(f"TIF reader initialized: {self.data.size} pixels in memory")
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get raster bounding box.
        
        Returns:
            (xmin, ymin, xmax, ymax)
        """
        bounds = self.src.bounds
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    def get_crs(self) -> str:
        """Get coordinate reference system."""
        return str(self.src.crs)
    
    def get_resolution(self) -> Tuple[float, float]:
        """
        Get pixel resolution.
        
        Returns:
            (pixel_width, pixel_height) in CRS units
        """
        return (abs(self.src.transform.a), abs(self.src.transform.e))
    
    def _xy_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert UTM coordinates to raster array indices.
        
        Args:
            x: UTM X coordinate
            y: UTM Y coordinate
            
        Returns:
            (row, col) indices in raster array
        """
        # Use the affine transform to convert coordinates to pixel indices
        # rasterio.transform.Affine provides the ~(x, y) operator for this
        row, col = ~self.src.transform * (x, y)
        return int(round(row)), int(round(col))
    
    def _ij_to_xy(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert raster indices to UTM coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            (x, y) UTM coordinates
        """
        # Use the affine transform to convert pixel indices to coordinates
        x, y = self.src.transform * (col, row)
        return float(x), float(y)
    
    def get_elevation(self, x: float, y: float, interpolate=False) -> Optional[float]:
        """
        Get elevation at a point.
        
        Args:
            x: UTM X coordinate
            y: UTM Y coordinate
            interpolate: If True, use bilinear interpolation; if False, nearest neighbor
            
        Returns:
            Elevation value or None if outside bounds
        """
        # Check bounds
        bounds = self.get_bounds()
        if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            logger.debug(f"Point ({x:.0f}, {y:.0f}) outside raster bounds")
            return None
        
        try:
            row, col = self._xy_to_ij(x, y)
            
            # Check array bounds
            if not (0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]):
                logger.debug(f"Indices ({row}, {col}) outside array bounds")
                return None
            
            if not interpolate:
                # Nearest neighbor
                z = self.data[row, col]
                if np.isnan(z):
                    return None
                return float(z)
            else:
                # Bilinear interpolation
                return self._interpolate_bilinear(x, y, row, col)
        
        except Exception as e:
            logger.warning(f"Error getting elevation at ({x:.0f}, {y:.0f}): {e}")
            return None
    
    def _interpolate_bilinear(self, x: float, y: float, row: int, col: int) -> Optional[float]:
        """
        Bilinear interpolation around a point.
        
        Args:
            x, y: UTM coordinates
            row, col: Raster indices
            
        Returns:
            Interpolated elevation or None
        """
        # Get grid resolution
        res_x, res_y = self.get_resolution()
        
        # Get coordinates of surrounding pixels
        x_pixel, y_pixel = self._ij_to_xy(row, col)
        
        # Interpolation weights
        wx = (x - x_pixel) / res_x + 0.5
        wy = (y - y_pixel) / res_y + 0.5
        
        # Ensure weights are in [0, 1]
        wx = np.clip(wx, 0, 1)
        wy = np.clip(wy, 0, 1)
        
        # Get 4 neighbors
        try:
            z00 = self.data[row, col]
            z10 = self.data[row, min(col + 1, self.data.shape[1] - 1)]
            z01 = self.data[min(row + 1, self.data.shape[0] - 1), col]
            z11 = self.data[min(row + 1, self.data.shape[0] - 1), min(col + 1, self.data.shape[1] - 1)]
            
            # Skip if any neighbor is NaN
            if np.isnan(z00) or np.isnan(z10) or np.isnan(z01) or np.isnan(z11):
                return None
            
            # Bilinear interpolation
            z = (z00 * (1 - wx) * (1 - wy) +
                 z10 * wx * (1 - wy) +
                 z01 * (1 - wx) * wy +
                 z11 * wx * wy)
            
            return float(z)
        
        except Exception as e:
            logger.warning(f"Interpolation error at ({row}, {col}): {e}")
            return None
    
    def extract_patch(self, x_center: float, y_center: float, 
                     patch_size_m: float) -> Optional[np.ndarray]:
        """
        Extract raster patch around a location.
        
        Args:
            x_center: Center X coordinate
            y_center: Center Y coordinate
            patch_size_m: Side length in meters
            
        Returns:
            2D array of elevation data, or None if outside bounds
        """
        # Check bounds
        bounds = self.get_bounds()
        if not (bounds[0] <= x_center <= bounds[2] and bounds[1] <= y_center <= bounds[3]):
            return None
        
        # Convert to raster coordinates
        row_c, col_c = self._xy_to_ij(x_center, y_center)
        
        # Calculate patch size in pixels
        res_x, res_y = self.get_resolution()
        half_pixels_x = int(patch_size_m / (2 * res_x))
        half_pixels_y = int(patch_size_m / (2 * res_y))
        
        # Extract patch
        row_min = max(0, row_c - half_pixels_y)
        row_max = min(self.data.shape[0], row_c + half_pixels_y)
        col_min = max(0, col_c - half_pixels_x)
        col_max = min(self.data.shape[1], col_c + half_pixels_x)
        
        patch = self.data[row_min:row_max, col_min:col_max]
        
        if patch.size == 0:
            return None
        
        return patch
    
    def get_stats(self) -> dict:
        """Get raster statistics."""
        return {
            'width': self.src.width,
            'height': self.src.height,
            'size_pixels': self.data.size,
            'resolution': self.get_resolution(),
            'bounds': self.get_bounds(),
            'z_min': float(np.nanmin(self.data)),
            'z_max': float(np.nanmax(self.data)),
            'z_mean': float(np.nanmean(self.data)),
            'z_std': float(np.nanstd(self.data)),
            'n_nodata': int(np.isnan(self.data).sum())
        }
    
    def close(self):
        """Close the raster file."""
        if hasattr(self, 'src'):
            self.src.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
